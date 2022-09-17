# Handling-Extremely-Large-Images
Loading Extremely Large Images e.g. 100.000 x 100.000 is an issue when need to load them, preprocess, and train a Deep Learning model. 
Also, common resizing tools like cv2 lead to huge info loss being slow and unable to resize in such scales. 
In this project we aim to provide a viable solution and adress these issues, accelerating the resizing procedure time for such images.


We provide the following open slide functions: **Slide_Open_Resize**, **Slide_Cut_off_Resize**, **Optimized_Slide_Pack**.

     
**Slide_Open_Resize**

                Has two modes. If the loading is image is too large for memory issues,
                it opens it with slide mode. Then, it automatically resizes every loaded image region tile,
                and final it conatenates all resized tiles together building the final image. If image has a viable size,
                it opens it, in classic cv2 mode.
                
**Slide_Cut_off_Resize**

                Always uses slide mode. 
                The main difference, comparing to previous approach, via our blank_tile_detector, 
                it can toss out useless blank areas and thus the important areas/objects of the 
                final image will have higher resolution!
                However, the distances and the initial locations between each object of the initial images are lost.
                This can be a slight information loss, if want to use the extracted image into a CNN model 
                for deep representation learning porpuses. 
                Thus, we extract also an optional "blank_map" for further use, which contain the locations of objects in a map form.
                
**Optimized_Slide_Pack**

                Always uses slide mode. It can be used for slide packs extraction given an initial huge image.
                Via our hand-crafted blank_tile_detector the final extracted slide pack will be composed by **ONLY** dense object areas 
                and not useless blank slides. This is significant when need to feed the slides into 
                a CNN model since it will obtain much area information.
                Comparing to other approaches, with this feed type, a CNN can focus on local object areas, 
                exploiting much more higher resolutions of the initial huge image. 
                However, the whole structure shape information is lost.
                Finally, it can also act (by some minor modifications), 
                as an _intelligent random crop generator for huge initial images, for data augmentation purposes, 
                since it will only add random crops of valuable dense object areas.
