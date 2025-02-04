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
