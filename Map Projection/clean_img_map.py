# combining into one piece
# 1. reconcile map and images by name
# 2. delete maps not having images
# 3. crop maps to locate tip pf the car in the same position on map image

#imports
import os
import pandas as pd
import cv2

#constants
PIXELS_FROM_MAP_BOTTOM = 210 #number of most low rows of map image to locate the car in
PIXELS_FROM_CENTRE_SIDEWISE = 10 #number of pixels to take to the sides off the centre fo locating the car shape

CROP_MAP_PIXELS_Y = 264
CROP_MAP_PIXELS_X = 240

PATH_MAP = 'C:/SelfDrive/Map Projection/map_img/'
PATH_IMG = 'C:/SelfDrive/Map Projection/img/'
PATH_MAP_CLEAN = 'C:/SelfDrive/Map Projection/map_clean/'

# read lists of maps and images
images = [f for f in os.listdir(PATH_IMG) if f.endswith(".png")]
maps = [f for f in os.listdir(PATH_MAP) if f.endswith(".png")]

im_df = pd.DataFrame(images)
map_df = pd.DataFrame(maps)

# find not synched files to be deleted
img_all = pd.merge(left=im_df,right=map_df,on=0,how='left',indicator=True)
img_extra = img_all[img_all['_merge'] == 'left_only'] 
img_extra.drop(columns=['_merge'],inplace=True)

map_all = pd.merge(left=map_df,right=im_df,on=0,how='left',indicator=True)
map_extra = map_all[map_all['_merge'] == 'left_only']  #.drop('_merge') 
map_extra.drop(columns=['_merge'],inplace=True)

for f in img_extra[0]:
    os.remove(PATH_IMG+str(f))
for f in map_extra[0]:
    os.remove(PATH_MAP+str(f))

print('... deleted:',len(map_extra),' extra maps and ',len(img_extra),' images.')

# This section finds row index when green tip of the car is located on the map
# so it could be used to crop all images to achive consistent location of the map

# loop through maps to get a range of possible tip location 
maps = [f for f in os.listdir(PATH_MAP) if f.endswith(".png")]
tip_min = 1000
tip_max = 0
for m in maps:
    sample_map = cv2.imread(PATH_MAP+m,cv2.IMREAD_COLOR)
    # get a section of image with the car's outline
    slice_map = sample_map
    #finding green colour in image
    hsv_img = cv2.cvtColor(slice_map, cv2.COLOR_BGR2HSV)
    lower_range = (40, 40, 40) # lower range of green
    upper_range = (70, 255, 255) # upper range of green
    mask = cv2.inRange(hsv_img, lower_range, upper_range)

    region_to_look_y = slice_map.shape[0]-PIXELS_FROM_MAP_BOTTOM
    region_to_look_x1 = int(slice_map.shape[1]/2-PIXELS_FROM_CENTRE_SIDEWISE)
    region_to_look_x2 = int(slice_map.shape[1]/2+PIXELS_FROM_CENTRE_SIDEWISE)

    crop = mask[region_to_look_y:, region_to_look_x1:region_to_look_x2]
    point_loc = 0
    for i in range(crop.shape[0]):
        if crop[i,].sum() > 0 and point_loc==0:
            point_loc = i
    row_of_arrow = region_to_look_y + point_loc
    if tip_min > row_of_arrow:
        tip_min = row_of_arrow
    if tip_max < row_of_arrow:
        tip_max = row_of_arrow
    crop_x = int(slice_map.shape[1]/2-CROP_MAP_PIXELS_X/2)
    crop_y = row_of_arrow + 20 - CROP_MAP_PIXELS_Y
    final_crop = sample_map[crop_y:crop_y+CROP_MAP_PIXELS_Y,crop_x:crop_x+CROP_MAP_PIXELS_X]
    cv2.imwrite(PATH_MAP_CLEAN+m, final_crop)
    
print("Min index of row in original map of arrow tip is: ",tip_min)
print("Max index of row in original map of arrow tip is: ",tip_max)