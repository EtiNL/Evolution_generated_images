from particles_class import Particles
import numpy as np
import cv2
import time
from PIL import Image
from score import loss

def main(target_img_path, number_gen, intermediary_show = False):
    start = time.time()
    targetIm = np.array(Image.open(target_img_path))
    targetIm = cv2.resize(targetIm, (targetIm.shape[1]//4,targetIm.shape[0]//4))
    genIm = np.zeros_like(targetIm)*255
    loss_val = loss(targetIm, genIm)
    ds_coef = 8

    for i in range(number_gen):
        radius = []

        gen = Particles(50, 1, targetIm, genIm, ds_coef)
        cv2.imshow(f"Target", cv2.cvtColor(targetIm, cv2.COLOR_RGB2BGR))
        # if intermediary_show: cv2.imshow(f"gen{i}_init", cv2.cvtColor(gen.draw_particules(np.zeros_like(targetIm), targetIm), cv2.COLOR_RGB2BGR))
        gen.keep_n_best(n=10)
        # if intermediary_show: cv2.imshow(f"gen{i}_keep", cv2.cvtColor(gen.draw_particules(np.zeros_like(targetIm), targetIm), cv2.COLOR_RGB2BGR))
        gen.generate_noise(0.1)
        # if intermediary_show: cv2.imshow(f"gen{i}_noise", cv2.cvtColor(gen.draw_particules(np.zeros_like(targetIm), targetIm), cv2.COLOR_RGB2BGR))
        gen.keep_n_best()
        radius.append(gen.radius[0])
        loss_val, genIm = gen.draw_particules(loss_val)
        if i%100 == 0 or intermediary_show:
            stop = time.time()
            print(f'gen_{i} loss = ', loss_val, f'___ time/10_gen: {int((stop-start)//60)}m {round((stop-start)%60)}s')
            print(f'Mean radius drawn = {np.mean(radius)}')
            if np.mean(np.array(radius)) < 5 and ds_coef>2:
                ds_coef = int(ds_coef//2)
            radius = []
            cv2.imshow(f"gen{i}", cv2.cvtColor(genIm, cv2.COLOR_RGB2BGR))
            start = time.time()
        elif i%10==0 and not intermediary_show: print(f'gen_{i} loss = ', loss_val)




    cv2.imshow(f"gen{i}", cv2.cvtColor(genIm, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)

if __name__=='__main__':
    main('/content/drive/MyDrive/La_force_des_vagues.JPG', 10000, intermediary_show = False)
