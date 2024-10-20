# LiDAR-Pose-Detection

### Setup

1. Install Python

   Visit the official Python website: https://www.python.org/downloads/
   Download the latest version for your operating system (Windows, macOS, or Linux).
   During installation:
   Check the box “Add Python to PATH” (important for command-line use).
   Use default settings for the rest of the installation.
   After installation, open a terminal or command prompt and type:

   ```shell
   python --version
   ```

2. Download the dpt_hybrid_nyu-2ce69ec7.pt model and place in ./utils/dpt/weights/ directory

- Monodepth finetuned on NYUv2: [dpt_hybrid_nyu-2ce69ec7.pt](https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid_nyu-2ce69ec7.pt) [Mirror](https://drive.google.com/file/d/1NjiFw1Z9lUAfTPZu4uQ9gourVwvmd58O/view?usp=sharing)

3. Set up dependencies:

   ```shell
   pip install -r requirements.txt
   ```

4. (Option 1) To test application on sample images choose sample image from ./sample_images/ directory and alter main_image.py file on line 34 as follows:
   example: ./sample_images/4.png

   ```py
   frame = cv2.imread("./sample_images/4.png")
   ```

   then run

   ```shell
   streamlit run main_image.py
   ```

5. (Option 2) To test application on live webcam footage choose run

   ```shell
   streamlit run main.py
   ```

### Citations

```
@article{Ranftl2021,
	author    = {Ren\'{e} Ranftl and Alexey Bochkovskiy and Vladlen Koltun},
	title     = {Vision Transformers for Dense Prediction},
	journal   = {ArXiv preprint},
	year      = {2021},
}
```

```
@article{Ranftl2020,
	author    = {Ren\'{e} Ranftl and Katrin Lasinger and David Hafner and Konrad Schindler and Vladlen Koltun},
	title     = {Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer},
	journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
	year      = {2020},
}
```

### License

MIT License
