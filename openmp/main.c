#include <omp.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <time.h>

typedef unsigned char uchar;

void save(cv::Vec3b **pixels, int height, int width, const std::string &name,
          const std::string &type) {
  cv::Mat image(height, width, CV_8UC3);
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      cv::Vec3b pixel = pixels[i][j];
      image.at<cv::Vec3b>(i, j) = pixel;
    }
  }
  std::string path = type + name;
  cv::imwrite(path, image);
  std::cout << "Image saved to: " << path << std::endl;
}

void bicubic_interpolation(cv::Vec3b **bpixels, cv::Vec3b **pixels, int height,
                           int width, int threadsNumber) {
#pragma omp sections
  {
#pragma omp section
    {
      int ww = (width - 1) >> 1;
#pragma omp parallel for schedule(static)                                      \
    shared(height, width, pixels, bpixels, ww)
      for (int i = 0; i < height;
           i++) { // I dont know what on earth this does but it works ig
        bpixels[i][0] = pixels[i >> 1][0]; // only for j == 0 or j == width - 1
        bpixels[i][width - 1] = pixels[i >> 1][ww];
      }
    }
#pragma omp section
    {
      int hh = (height - 1) >> 1;
#pragma omp parallel for schedule(static)                                      \
    shared(height, width, pixels, bpixels, hh)
      for (int j = 0; j < height; j++) { // i == 0 or i == height - 1
        bpixels[0][j] = pixels[0][j >> 1];
        bpixels[height - 1][j] = pixels[hh][j >> 1];
      }
    }
#pragma omp section
    {
#pragma omp parallel for schedule(static) shared(height, width, pixels, bpixels)
      for (int i = 1; i < height - 1; i++) {
        for (int j = 1; j < width - 1; j++) {
          unsigned int suma[] = {0, 0, 0};
          for (int k = -1; k < 2; k++) {
            int ik = (i + k) >> 1;
            for (int l = -1; l < 2; l++) {
              int jl = (j + l) >> 1;
              for (int m = 0; m < 3; m++) {
                suma[m] += pixels[ik][jl][m];
              }
            }
          }

          for (int m = 0; m < 3; m++) {
            bpixels[i][j][m] = (unsigned char)(suma[m] / 9);
          }
        }
      }
    }
  }
}

void write_time_to_csv(const char *image_path, double time_taken) {
  FILE *fp = fopen("openmp_time_measurements.csv",
                   "a"); // Open the CSV file in append mode
  if (fp == NULL) {
    printf("Error opening file\n");
    return;
  }

  // Write the image path and time taken to the CSV file
  fprintf(fp, "%s,%.6f\n", image_path, time_taken);

  fclose(fp); // Close the file
}

int main(int argc, char **argv) {

  if (argc != 2) {
    printf("Usage: %s <image_path>\n", argv[0]);
    return -1;
  }

  cv::Mat image = cv::imread(argv[1], cv::IMREAD_COLOR);

  if (image.empty()) {
    printf("Could not open or find the image\n");
    return -1;
  }

  int height = image.rows;
  int width = image.cols;
  int num_threads = omp_get_num_procs();

  cv::Vec3b **pixels = new cv::Vec3b *[height];
  cv::Vec3b **bicubic_pixels = new cv::Vec3b *[2 * height];

  for (int i = 0; i < height; i++) {
    pixels[i] = new cv::Vec3b[width];
    bicubic_pixels[i] = new cv::Vec3b[2 * width];
    bicubic_pixels[height + i] = new cv::Vec3b[2 * width];
  }
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      pixels[i][j] = image.at<cv::Vec3b>(i, j);
    }
  }
  printf("Started process with %d threads\n", num_threads);

  double start = omp_get_wtime();
  bicubic_interpolation(bicubic_pixels, pixels, height * 2, width * 2,
                        num_threads);
  double end = omp_get_wtime();

  // printf("Time taken: %fs\n", end - start);
  write_time_to_csv(argv[1], end - start);

  // save(bicubic_pixels, 2 * height, 2 * width, argv[1],
  // "../output/openmp_bicubic_");

  for (int i = 0; i < height; i++) {
    delete[] pixels[i];
    delete[] bicubic_pixels[i];
    delete[] bicubic_pixels[height + i];
  }
  delete[] pixels;
  delete[] bicubic_pixels;

  return 0;
}
