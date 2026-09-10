# 🚀 Astro-Pi ISS Speed Estimator (ESA Challenge)

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv)
![Raspberry Pi](https://img.shields.io/badge/Hardware-Raspberry%20Pi-red?logo=raspberrypi)
![Status](https://img.shields.io/badge/Mission-Executed%20on%20ISS-orange)

An edge computing application designed to calculate the orbital speed of the International Space Station (ISS) in real time during a 10-minute window, using ground-facing camera imagery and telemetry data.

Executed aboard the ISS as part of the **European Space Agency (ESA) Astro Pi Challenge**.

## 🛠️ Tech Stack & Key Features

- **Computer Vision & Speed Estimation:**
    - **Feature Matching (ORB + BruteForce):** Tracks landmarks between consecutive Earth images to derive displacement.
    - **Exif Metadata & Geolocation:** Extracts GPS telemetry and calculates geodesic distance via the **Haversine formula**.
- **Sensor Telemetry & IoT:**
    - Real-time logging of IMU (Gyroscope, Accelerometer, Magnetometer) and environmental metrics (Temperature, Pressure, Humidity) via **Sense HAT**.
- **Data Engineering & Analytics:**
    - Outlier removal pipeline using Interquartile Range (IQR).
    - Automated generation of CSV reports and diagnostic plots (**Pandas**, **Matplotlib**).

## 📐 Pipeline Overview

1. **Acquisition:** Captures high-resolution ground images with PiCamera + extracts Exif GPS tags.
2. **Analysis:** Computes speed dual-way (ORB feature tracking vs. Geodesic coordinate Delta).
3. **Filtering:** Applies IQR filtering to eliminate visual artifacts/cloud interference.
4. **Export:** Generates real-time plots, tracks ISS trajectory on world maps, and dumps logs to file.

## 📊 Results & ISS Telemetry

| Metric                | Source / Method                                  |
| :-------------------- | :----------------------------------------------- |
| **Primary Method**    | OpenCV ORB Feature Distance / Time Delta         |
| **Validation Method** | Exif GPS Coordinates + Haversine Formula         |
| **Hardware Target**   | Raspberry Pi Flight Unit + Sense HAT + Pi Camera |

_(Include here an image of your generated plots or ISS track map)_
`![ISS Tracking Map](assets/stationsTracking.png)`

## 👥 Authors & Context

Project developed by **Team Astro Elite** (Belgium) for the ESA Astro Pi Challenge.

- **School:** Institut Saint-François de Sales (Ath, Belgium)
- **Team:** Florian Berte, Thibaut Dudart, Rafaël Ravry
