- Project structure (Extract the .zip member face into the dataset folder):
PCA_LDA_face_detection/
├── eigen_opencv.cpp
├── fisher_opencv.cpp
├── fisher_opencv.py
├── create_csv.py
├── create_modified_faces.py
├── offset_face.py
├── dataset/
   ├──Members Face
      ├──An
      ├──Edward
      ├──Ereston
      ├──Haechan
      ├──Ivo
      ├──Nam
      ├──member.txt
      ├──modified
         ├──An
         ├──Edward
         ├──Ereston
         ├──Haechan
         ├──Ivo
         ├──Nam
         ├──member_modified.txt
├── haarcascade_frontalface_default.xml
├── lbfmodel.yaml
├── res10_300x300_ssd_iter_140000.caffemodel
├── README.md

- Run the file in order (the current directory is PCA_LDA_face_detection):
python .\create_csv.py ".\dataset\Members Face"            ===>    get the file path for each image 
python .\create_modified_faces.py                          ===>    crop the image into a 200x200 face image, the new folder modified will appear
python .\create_csv.py ".\dataset\Members Face\modified"   ===>    get the file path for each modified image
python .\fisher_opencv.py "dataset\Members Face\modified\member_modified.txt" "output\member\fisher"    ===>    Run live face recognition and generate an output folder for visualization of each mean faces

- Google collab notebook is provided for online testing: EE4228_Assignment_1.ipynb
