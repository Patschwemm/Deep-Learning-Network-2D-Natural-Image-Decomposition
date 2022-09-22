from struct import unpack
import torch
import torch.nn as nn
from losses import *
from preprocess_dataset import mask_rgb_imgs
import main_network
from tqdm import tqdm


def train_model(
    epoch, loader, extract_fn, model, num_epoch_fixed_prob, batch_size, optimizer, device,
):
    model = model.to(device)
    all_cov_loss = []
    all_cons_loss_rect, all_cons_loss_circ = [], []
    for ep in range(epoch):

        cov_loss= []
        cons_circ, cons_rect = [], []
        for data, target in (pbar := tqdm((loader), position=0, leave=True)):

            # custom extract function for different datasets
            data, mask = extract_fn(data, target, device)

            # output is rect circ triangle torch tensors
            output = model(data)
            # get dictionary to sort outputs if any number of prim is set to 0
            num_dict = model.num_dict

            # unpack output according to numbers of primitives
            r, rp, r_rot, c, cp = unpack_model_output(num_dict, output)

            z_r = (
                torch.bernoulli(rp)
                if num_epoch_fixed_prob <= ep and rp != None
                else None
            )
            z_c = (
                torch.bernoulli(cp)
                if num_epoch_fixed_prob <= ep and cp != None
                else None
            )

            if z_c != None and z_r != None:
                z_all = torch.cat((z_r, z_c), dim=1)
            elif z_c != None and z_r == None:
                z_all = z_c
            else:
                z_all = z_r
            union_cov_loss = coverage_loss_all(r, r_rot, c, mask, z_all)

            # compute losses for particular primitives
            cons_loss_rect, prob_loss_rect = compute_cons_prob_loss_primtype(
                r, rp, r_rot, z_r, mask, "Rectangle", union_cov_loss, num_epoch_fixed_prob, ep, batch_size, device
            )
            cons_loss_circ, prob_loss_circ = compute_cons_prob_loss_primtype(
                c, cp, r_rot, z_c, mask, "Circle", union_cov_loss, num_epoch_fixed_prob, ep, batch_size, device
            )

            # add all losses together
            all_prim_loss = (
                union_cov_loss
                + cons_loss_circ
                + prob_loss_circ
                + cons_loss_rect
                + prob_loss_rect
            )

            optimizer.zero_grad()
            all_prim_loss.mean().backward()
            optimizer.step()

            # update tqdm loop
            cov_loss.append(union_cov_loss.mean())
            cons_rect.append(cons_loss_rect.mean())
            cons_circ.append(cons_loss_circ.mean())

            pbar.set_postfix(
                loss=(
                    "all losses: %.05f union cov loss: %.05f cons_loss_rect %.05f cons_loss circle %.05f  "
                    % (
                        sum(cov_loss) / len(cov_loss)
                        + sum(cons_rect) / len(cons_rect)
                        + sum(cons_circ) / len(cons_circ),
                        sum(cov_loss) / len(cov_loss),
                        sum(cons_rect) / len(cons_rect),
                        sum(cons_circ) / len(cons_circ)
                    )
                )
            )

        all_cov_loss.append(cov_loss)
        all_cons_loss_rect.append(cons_rect)
        all_cons_loss_circ.append(cons_circ)

    model.to(device="cpu")
    torch.cuda.empty_cache()

    return all_cov_loss, all_cons_loss_rect, all_cons_loss_circ

def extract_synthetic_data(data, field, device):
    return data.unsqueeze(1).to(device), data.to(device)

def extract_pet_data(data, targets, device):
    #extract background from natural image
    masked_rgb = mask_rgb_imgs(data, targets)
    masked_rgb = masked_rgb.float()

    targets = torch.permute(targets, (2, 1, 0, 3, 4))
    targets = targets.float()
    # train data but with only the binary mask
    x_targets = torch.permute(targets, (1, 2, 0, 3, 4))[0]
    mask = x_targets.squeeze()
    return masked_rgb.to(device), mask.to(device)

def extract_mnist(data, targets, device):
    return data.to(device), data.squeeze().to(device)

def extract_cad(image, mask, device):
    return image.to(device), mask.squeeze().to(device)

def unpack_model_output(prim_dict, output):
    # conditions if num of primitives is greater than 0 to unpack output variables
    if prim_dict["Rectangle"] > 0:
        r, rp, r_rot = output["Rectangle"]
    else:
        r, rp, r_rot = None, None, None
    if prim_dict["Circle"] > 0:
        c, cp = output["Circle"]
    else:
        c, cp = None, None

    return r, rp, r_rot, c, cp

def compute_prob_loss_primitive(num_epoch_fixed_prob, ep, loss_prim, p, z):
    # after a fixed amout of time compute probabilities
    if num_epoch_fixed_prob <= ep:
        with torch.no_grad():
            # reward low loss_prim and low number of primitives selected
            loss_prim = loss_prim.unsqueeze(-1)
            # the scalar factor selects how many rectangles are importan (0 for no rectangle, 1 for all)
            reward = -(loss_prim - loss_prim.mean()) - p * 0.5
            # compute loss_prim from reward
        log_probs = torch.where(z.bool(), p, 1 - p).log()
        prob_loss = (-log_probs * reward).mean(-1)
    else:
        prob_loss = 0.0
    return prob_loss


def compute_cons_prob_loss_primtype(
    p, p_prob, p_rot, z_p, mask, mode, cov_loss_prim, num_epoch_fixed_prob, ep, batch_size, device
):

    # computes coverage, consistency and probablility loss for each primitive type
    if p_prob != None and p != None:
        cons_loss_prim = consistency_loss(p, p_rot, mask, z=z_p, mode=mode)
        prob_loss_rect = compute_prob_loss_primitive(
            num_epoch_fixed_prob, ep, (cov_loss_prim + cons_loss_prim), p_prob, z_p
        )
        return cons_loss_prim, prob_loss_rect
    else:
        zero_tensor = torch.zeros(size=[batch_size], device=device)
        return zero_tensor, zero_tensor

def return_selected_prims(prim, prim_prob, threshold, selected=True):
    # function returns the selected values with selected set to true
    # otherwise returns the unselected primitives
    if prim == None:
        return None
    else:
        z_prim = (prim_prob > threshold).reshape(prim.size(0), -1)
        print(z_prim.shape)
        selected = torch.masked_select(prim, z_prim)
        print(selected)
        return prim[z_prim] if selected else prim[~z_prim]

def IoU(
    r: torch.Tensor,
    r_rot: torch.Tensor, 
    rp: torch.Tensor,
    c: torch.Tensor,
    cp: torch.Tensor,
    prob_threshold: float,
    mask: torch.Tensor,
    device: torch.device,
    mode: str="batch"
):
    # get batch size, primitive count and compute distance field for rectangles and circles
    # rectangle primitive count and distance field extraction
    if r != None:
        b = r.size(0)
        p_rect= r.size(1)
        # compute distance field
        d_rect = compute_rotated_rectangle_distance_field(r.reshape(-1, 2, 2), r_rot, torch.tensor([mask.size(-2), mask.size(-1)]))
        # set distance field to 0 for the mask, reshape for batch shape
        d_rect = (d_rect < 1).reshape(b, p_rect, mask.size(-1), mask.size(-2))
        # apply the probability mask to the unselected primitives
        d_rect = d_rect.masked_fill(rp.reshape(b, p_rect, 1, 1) < prob_threshold, False)
        print(dist_threshold)
        
        d_rect_union = d_rect.any(dim=1)
        _, ax = plt.subplots(1, 2, figsize=(15, 15))
        ax[0].imshow(mask[5].squeeze(), cmap="gray")
        ax[1].imshow(d_rect_union[5].squeeze(), cmap="gray")
        plt.savefig(f"pictures/d_rect_union.png")
        plt.close()

        
    else:
        p_rect = 0
        d_rect = torch.tensor((), device=c.device)
    # circle primitive count and distance field extraction
    if c != None:
        b = c.size(0)
        p_circ= c.size(1)
        # compute distance field
        d_circ = compute_circle_distance_field(c.reshape(-1, 1, 3), torch.tensor([mask.size(-1), mask.size(-2)]))
        # set distance field to 0 for the mask, reshape for batch shape
        d_circ = (d_circ == 0).reshape(b, p_circ, mask.size(-1), mask.size(-2))
        # apply the probability mask to the unselected primitives
        d_circ = d_circ.masked_fill(cp.reshape(b, p_circ, 1, 1) < prob_threshold, False)
    else:
        p_circ = 0
        d_circ = torch.tensor((), device=r.device)

    # concatinate the masks in the first dimension
    union_mask = torch.cat((d_rect, d_circ), dim = 1) 
    # logical or of the masks to extract the union mask
    union_mask = union_mask.any(dim=1)

    intersec = (union_mask * mask).sum()
    union = torch.clamp((union_mask + mask), min=0, max=1).sum()
 
    # _, ax = plt.subplots(1, 2, figsize=(15, 15))
    # ax[0].imshow(mask[0].squeeze(), cmap="gray")
    # ax[1].imshow(union_mask[1], cmap="gray")
    # plt.savefig(f"pictures/intersec_union.png")
    # plt.close()

    if mode == "elementwise":
        return intersec / union
    if mode == "batch":
        return (intersec / union).mean()
    else:
        return print(
            "IoU ftn: No mode has been selected, please select mode = 'elementwise' or 'batch'"
        )