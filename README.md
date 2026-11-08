# Big Data Analytics Project

This repository contains a Big Data Analytics project that analyzes the Amazon Video Games review dataset using Apache Spark. 

> **Note:** The dataset is **not included** in this repository because of its large size. Please download it separately and place it inside the `data/` directory.

---

## Project Structure

```
.
├── data/                     # Dataset (ignored by Git)
│   ├── metadata.json
│   └── reviews.json
│
├── notebooks/
│   ├── 01_data_loading.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_helpfulness.ipynb
│   ├── 04_graph.ipynb
│   └── outputs/              # Generated figures and CSV results
│
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Technologies Used

- Apache Spark (PySpark)
- GraphFrames
- NetworkX
- Docker & Docker Compose
- Jupyter Notebook
- Python
- Pandas
- Matplotlib

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<repository-name>.git
cd <repository-name>
```

### 2. Download the dataset

Download the Amazon Video Games dataset and place the files inside:

```
data/
├── metadata.json
└── reviews.json
```

The `data/` directory is excluded from Git because the dataset is too large.

### 3. Start the environment

```bash
docker-compose up --build
```

### 4. Open Jupyter Notebook

After the containers start, open the Jupyter Notebook URL provided in the terminal and run the notebooks in the following order:

1. `01_data_loading.ipynb`
2. `02_data_preprocessing.ipynb`
3. `03_helpfulness.ipynb`
4. `04_graph.ipynb`

---

## Notes

- The dataset is intentionally excluded from this repository (`.gitignore`).
- Ensure the dataset files are placed inside the `data/` directory before running the notebooks.
- Run the notebooks sequentially, as later notebooks depend on outputs from earlier stages.

---
