CMPSC 457
Allen Brubaker
ajb5377@psu.edu
4/25/2010
Problem 4 - Readme

(In case you think I cheated and used last years file, I didn't :).  I completely rewrote the whole thing and utilized an abstract shape class with pure virtual functions overloaded by sphere and triangle.  I also used vectors instead of arrays.  Code is cleaner, faster, and simpler.. and works now for spherical lights! (300 lines less code))


FEATURES

Features implemented are as follows:
Ray-tracing with spheres, triangles, box filtering, texture mapping, shadows, reflection, lighting that assumes spherical or triangular shapes, and arbitrary view point.

CONFIGURATION

The configuration of the settings file, spheres, and triangle files have bee n slightly modified from the assignment description.

Command Format
-----------------
ambient r g b   - Specifies ambient light of scene
pixelsamples n  - n*n is the number of samples to trace thru each pixel.
lightsamples n  - n*n is the number of samples to take of each light src.
depth n         - n is the recursion depth for specular surfaces.

Spheres Format
-----------------
center.x center.y center.z radius [diffuse | texture | light] [diffuse.x diffuse.y diffuse.z | textureFileName.ppm | emit.x emit.y emit.z]

Triangles Format
-----------------
x0 y0 z0 u0 v0 x1 y1 z1 u1 v1 x2 y2 z2 u2 v2 [diffuse, texture, light] [diffuse.x diffuse.y diffuse.z, textureFileName.ppm, emit.x emit.y emit.z]
    

INSTRUCTIONS

1.  Navigate to http://cs.hbg.psu.edu/~ajb5377/.
2.  Once on webpage, download the two .ppm texture files and put into this directory.
2.  Type in terminal: /usr/sfw/bin/g++ -ansi -pedantic prog4.cpp -o prog4
3.  Type in terminal: ./prog4 512 512 settings.txt output.ppm
4.  View output.ppm for final image.  


