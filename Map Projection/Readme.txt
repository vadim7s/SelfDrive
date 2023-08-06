This section is to try getting a close map from where the car is by leveraging
 the simlulator "cheats". The map can be used for:
1. Making a visual - overlaying the map on display (like in a Tesla)
2. Making modelas from RGB camera to the map
3. Using the map for planner

Note on colours: No_rendering_mode.py must be using some external source
to get the actual map drawing. I tried changing all colour constants to red and
it only impacts what is drawn around the map. 
Conclusion: external functionality/library is used to draw the actual map