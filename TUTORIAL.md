# Sticky-traps
Sticky-traps is a python file which comes as an example in the ImgPheno package. It takes pictures of sticky traps with trapped insects as input and returns the number of insects counted on the traps as output. 

A YAML file ("sticky-traps.yml") is read into "sticky-traps.py" to specify information about the traps, such as height and width. 

## Running on command line
Install with:

    git clone https://github.com/naturalis/imgpheno.git
    cd imgpheno
    sudo python setup.py install
    
After installation of the ImgPheno package and the requirements sticky-traps.py (imgpheno/examples/) is now able to be executed on the command line. 

The pictures of the traps that are used for analysis need to be stored in the images folder (imgpheno/examples/images/sticky-traps/). 
To provide the right information about the traps that are used for analysis the sticky-traps.yml (imgpheno/examples/) file needs to be adjusted:

 - Set the height and width of the traps, in mm.
 - Set if edges of the traps need to be cropped.
 - If edges need to be cropped, set the size of the area that needs to be cropped, in mm.
 - Set the colour of the traps, in HSV.
 - Set an output file if the results need to be saved in a file (.txt, .csv).

Be aware that the traps need to be uniform in colour (without markings, lines), uniform in size, and have a black or complete white background with, preferably, no reflections.
The YAML file has a comprehensive explanation for each of the to be adjusted settings.

After adjusting the settings the sticky-traps python file is executed by entering the examples directory with:

    cd examples

and then entering the command:

    python sticky-traps.py

The images are read into the sticky-traps python file and insects are counted, resulting in the total number of insects, the number of insects that are smaller than 4mm, the number of insects that are between 4 and 10mm, and the number of insects that are larger than 10mm. If a file is specified as output file it will be stored in examples. 

If an image cannot be analysed make sure to remake a photograph of the trap and it:
  
 - Contains the complete trap,
 - Is taken right above the trap,
 - Has a very dark, or completely white background, 
 - Has a high resolution and is clear enough to distinguish the smallest insects.