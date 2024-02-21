from particles_class import Particles
import numpy as np
import cv2
import time
from PIL import Image
from score import loss
from skimage.filters.rank import entropy
from skimage.morphology import disk

def main(target_img_path, number_gen, intermediary_show = False):
    start = time.time()
    targetIm = np.array(Image.open(target_img_path))
    targetIm = cv2.resize(targetIm, (targetIm.shape[1]//4,targetIm.shape[0]//4))
    genIm = np.zeros_like(targetIm)
    loss_val = loss(targetIm, genIm)
    ds_coef = 4

    for i in range(number_gen):
        radius = []

        gen = Particles(30, 1, targetIm, genIm, ds_coef)
        cv2.imshow(f"Target", cv2.cvtColor(targetIm, cv2.COLOR_RGB2BGR))
        # if intermediary_show: cv2.imshow(f"gen{i}_init", cv2.cvtColor(gen.draw_particules(np.zeros_like(targetIm), targetIm), cv2.COLOR_RGB2BGR))
        gen.keep_n_best(n=3)
        # if intermediary_show: cv2.imshow(f"gen{i}_keep", cv2.cvtColor(gen.draw_particules(np.zeros_like(targetIm), targetIm), cv2.COLOR_RGB2BGR))
        gen.generate_noise(0.1)
        # if intermediary_show: cv2.imshow(f"gen{i}_noise", cv2.cvtColor(gen.draw_particules(np.zeros_like(targetIm), targetIm), cv2.COLOR_RGB2BGR))
        gen.keep_n_best()
        radius.append(gen.radius[0])
        loss_val, genIm = gen.draw_particules(loss_val)
        if i%10==0:
            print(f'Mean radius drawn = {np.mean(radius)}')
            if np.mean(np.array(radius)) < 3 and ds_coef>2:
                ds_coef = int(ds_coef//2)
                radius = []
                print('__ upsampling __')
        if (i%100 == 0 and i!=0) or intermediary_show:
            stop = time.time()
            print(f'gen_{i} loss = ', loss_val, f'___ time/10_gen: {int((stop-start)//60)}m {round((stop-start)%60)}s')

            depth_map = np.sum(np.abs(targetIm - genIm), axis = 2)/3
            # print(( np.sum(targetIm, axis = 2).reshape((targetIm.shape[0], targetIm.shape[1], 1))).shape, targetIm)
            entropy_map = entropy(depth_map.astype(np.uint8), disk(10))
            max_entr, min_entr = np.max(entropy_map), np.min(entropy_map)
            entropy_map = (entropy_map - min_entr)/(max_entr - min_entr)*255

            # max_depth, min_depth = np.max(depth_map), np.min(depth_map)
            # depth_map = (depth_map - min_depth)/(max_depth - min_depth)*255

            rows_rgb, cols_rgb, channels = genIm.shape
            rows_gray, cols_gray = entropy_map.shape

            rows_comb = max(rows_rgb, rows_gray)
            cols_comb = cols_rgb + cols_gray
            comb = np.zeros(shape=(rows_comb, cols_comb, channels), dtype=np.uint8)

            comb[:rows_rgb, :cols_rgb] = genIm
            comb[:rows_gray, cols_rgb:] = entropy_map[:, :, None]
            cv2.imshow(f"gen{i}", cv2.cvtColor(comb, cv2.COLOR_RGB2BGR))


            start = time.time()
        elif i%10==0 and not intermediary_show: print(f'gen_{i} loss = ', loss_val)




    cv2.imshow(f"gen{i}", cv2.cvtColor(genIm, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)

if __name__=='__main__':
    main('raw_data/point.jpg', 1000, intermediary_show = False)
