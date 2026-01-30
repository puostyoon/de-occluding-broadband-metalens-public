#!/usr/bin/env python
import argparse
import torch
from .layers import v2_positional_encoding as v2

# def main():
   
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--inverse",   action="store_true",  help="Inverse the tone curve")
#     parser.add_argument("--no-grad",   action="store_false", dest="use_grad",     help="Disable gradients")
#     parser.add_argument("--no-hist",   action="store_false", dest="use_hist",     help="Disable soft histograms")
#     parser.add_argument("--no-satmask", action="store_false", dest="use_satmask", help="Disable saturation mask")
#     args = parser.parse_args()
    
#     device = torch.device('cuda:3')
#     model = ParamLocal(args) # Model declaration
#     model.to(device)
#     in_img = torch.zeros(8,1,256,256).to(device) # Input image
#     out_img = model(in_img) # Output image 
    
#     #print(1)
#     import pdb;pdb.set_trace() # 'q': end process


class ParamLocal(torch.nn.Module):
    """ Estimate tone curve with various methods. """
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.feat_channels = ((1 + 2 + 4+8+16) + 1)*3 +19 # pixel, gradient, soft histogram, over-exposed mask (+19: 19 positional encoding (pos_x, pos_y))
        if not args.use_grad:
            self.feat_channels -= 2*3 ##
        if not args.use_hist:
            self.feat_channels -= (4+8+16)*3  # no soft histograms

        self.main_channels = 64
        self.localnet = v2.LocalNet(self.feat_channels, self.main_channels, num_block=2, num_scale=2)

    def get_input_features(self, x: torch.Tensor):

        positional_encoding_feature = (x[:,-19:, ...]) # (N, 2, H, W) last 19 channels are positional encoding
        x = x[:,:3, ...] # Remove positional encoding part    

        # saturation mask
        if self.args.use_satmask:
            sat_mask = v2.over_exposed_mask(x)
        else:
            sat_mask = torch.zeros_like(x)      

        # gradient, histogram
        if self.args.use_grad or self.args.use_hist:
            content_features = v2.content_features(x, use_gradient=self.args.use_grad, histogram_bins=[4, 8, 16] if self.args.use_hist else [])
            return torch.cat([x, content_features, positional_encoding_feature, sat_mask], dim=-3)
        else: # no grad & no hist
            return torch.cat([x, positional_encoding_feature, sat_mask], dim=-3)

    def forward(self, in_img):

        in_feats = self.get_input_features(in_img)
        in_img = in_img[:,:3, ...] # Remove positional encoding part
        out_img = self.localnet(in_img, in_feats)
        return out_img


# if __name__ == "__main__":
#     main()