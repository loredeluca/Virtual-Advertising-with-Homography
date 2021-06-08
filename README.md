# Virtual-Advertising-with-Homography
This simple software uses homographies to display virtual advertisements in a realistic moving context. The results were obtained through the implementation of three different techniques starting from pre-recorded videos:
- **Identification of corners**: the user selects in the first frame the corners of the box where the virtual advertisement is inserted
![output](https://github.com/loredeluca/Virtual-Advertising-with-Homography/blob/main/results/g1.gif)
- **Use of Markers**: after inserting markers in the scene, the advertisement is placed above them.
![output](https://github.com/loredeluca/Virtual-Advertising-with-Homography/blob/main/results/g2.gif)
- **Identification of lines**: the software identifies the lines in the scene and uses them to insert advertising (to improved)
![output](https://github.com/loredeluca/Virtual-Advertising-with-Homography/blob/main/results/g3.gif)

### Installation and Use
To run the project, you will need [OpenCV](https://pypi.org/project/opencv-python/) and [Aruco](https://pypi.org/project/aruco/), then you can start script of the technique you prefer:
```sh
$ python3 Homography_Video.py
```
NB: To use the markers you can print the Aruco-markers in [this repository](https://github.com/loredeluca/Virtual-Advertising-with-Homography/tree/main/files/marker) or generate new ones [here](https://chev.me/arucogen/) (with dictionary = 6x6)
