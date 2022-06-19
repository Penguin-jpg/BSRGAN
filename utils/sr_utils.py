import os.path
import logging
import torch
import gc

from utils import utils_logger
from utils import utils_image as util

# from utils import utils_model
from models.network_rrdbnet import RRDBNet as net

# load model accroiding to model_name
def load_model(model_name, device):
    model_path = os.path.join("model_zoo", model_name + ".pth")

    if not os.path.isfile(model_path):
        print("model not found!")
        return

    if model_name == "BSRGANx2":  # 'BSRGANx2' for scale factor 2
        sf = 2
    else:
        sf = 4

    # define network and load model
    model = net(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=sf)  # define network
    model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=True)
    model.requires_grad_(False).eval().to(device)
    gc.collect()
    torch.cuda.empty_cache()

    return model


# image super resolution
def super_resolution(model, logger, device, testset_L="diffusion", save_results=True):
    utils_logger.logger_info("blind_sr_log", log_path="blind_sr_log.log")

    testsets = "testsets"  # fixed, set path of testsets
    logger.info("{:>16s} : {:<d}".format("GPU ID", torch.cuda.current_device()))

    L_path = os.path.join(testsets, testset_L)
    E_path = os.path.join("results", f"{testset_L}_results")
    util.mkdir(E_path)

    idx = 0

    for img in util.get_image_paths(L_path):

        # --------------------------------
        # (1) img_L
        # --------------------------------
        idx += 1
        img_name, ext = os.path.splitext(os.path.basename(img))
        logger.info(f"{idx: 4d} --> {img_name + ext:<s}")

        img_L = util.imread_uint(img, n_channels=3)
        img_L = util.uint2tensor4(img_L).to(device)

        # --------------------------------
        # (2) inference
        # --------------------------------
        img_E = model(img_L)

        # --------------------------------
        # (3) img_E
        # --------------------------------
        img_E = util.tensor2uint(img_E)
        if save_results:
            util.imsave(
                img_E,
                os.path.join(E_path, f"{img_name}_sr.png"),
            )

        del img_L  # delete from gpu to free memory
        torch.cuda.empty_cache()

    print("sr finished!")


# if __name__ == "__main__":
#     model_name = "BSRGAN"
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     logger = logging.getLogger("blind_sr_log")
#     model = load_model(model_name, device)
#     super_resolution(model, logger, device)
