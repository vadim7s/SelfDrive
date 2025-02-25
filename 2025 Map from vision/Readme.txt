This was to try Pytorch to train image to image CNN
which reads front facing camera and predicts top-down map view

Run in stages

1. Clean image and map folders

2. Generate images with map_grab_continue

3. Clean images with clean_img_map

4. Train model with torch_model

After a first attempt, I decided to build a more robust image generation engine with traffic, weather, multiple maps

as a result, generate training images1.py was created, but I failed to attache image count tracker to the very
complex object structure 

Then I tried ChatGPT and Deepseek's suggestions and they did not work well mostly traffic was not moving

then i decided to create separate python code with robust functions for:
1. traffic gen
2. map cutting
3. hero spawning with cameras 

so I could use ChatGPT structure to loop thrugh all maps

Outcome of map prediction was quite poor, however image generation in "generate training images2.py" is pretty good

It creates fron view RGB, front view semantic and Top down map, while having traffic and changing weather and time of day

A compromise there - you need to run it for one map at a time and re-run it a few times in that town as it slows down after few traffic re-starts

Next I wanted to train semantic segmentation model in Pytorch - "torch_model_3.py". 
Previously I trained a similar model in tensorflow and result was quite poor.

The result was quite good even after just 1 epoch on 200k image pairs

I tested the quality on a real Tesla footage - it was pretty good 