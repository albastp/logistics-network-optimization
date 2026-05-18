# Logistics Network Optimization

## Overview

This repository contains an end-to-end logistics optimization pipeline for distribution center placement and delivery route planning. The project combines data cleaning, feature engineering, exploratory analysis, clustering, and route optimization to propose better distribution center locations and lower delivery costs.

## Objective

Predict and optimize the logistics network by:
- cleaning and merging raw order, product, client, and payment data
- identifying national distribution center locations with K-Means
- designing a regional Nuevo León center layout with K-Medoids and road distances
- generating optimized delivery routes with cost and CO₂ estimates

## Key Features

- **Data pipeline:** raw file ingestion, cleanup, feature engineering, and master dataset construction
- **Exploratory data analysis:** frequency charts, histograms, boxplots, correlation matrices, ANOVA, and chi-square tests
- **National clustering:** K-Means clustering for 7 distribution centers across Mexico
- **Regional clustering:** Nuevo León clustering with K-Medoids and optional OSMnx road-network distances
- **Route optimization:** route assembly, baseline comparison, fuel cost, and CO₂ emissions calculations
- **Visualization:** interactive Folium maps for clusters and route comparisons

## Project Structure

```
Logistics Optimization/
├── data/                             # raw inputs and generated dataset
├── outputs/                          # generated figures, tables, and maps
├── src/
│   ├── data_processing/
│   │   ├── clean_raw_data.py         # raw Excel cleaning and validation
│   │   └── build_dataset.py          # dataset merge and feature engineering
│   ├── eda/
│   │   └── statistical_analysis.py   # EDA charting and statistical tests
│   ├── modeling/
│   │   ├── kmeans_national.py        # national K-Means clustering
│   │   └── kmedoids_nl.py            # Nuevo León K-Medoids clustering
│   ├── routing/
│   │   └── route_optimizer.py        # route generation and cost comparison
│   └── visualization/
│       └── maps.py                   # Folium map exports
├── main.py                           # pipeline entry point
├── requirements.txt                  # Python dependencies
└── README.md                         # project documentation
```
Note: Raw data files are not included in the repository.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

> `osmnx` and `folium` are optional for enhanced regional clustering and interactive maps.

## Usage

### Build the processed dataset from the raw files

```bash
python main.py --raw-data-dir data/raw --data data/df_final.csv --output outputs --steps all
```

### Run the full pipeline with an existing processed dataset

```bash
python main.py --data data/df_final.csv --output outputs --steps all
```

### Run specific pipeline steps

```bash
python main.py --data data/df_final.csv --output outputs --steps eda national regional routing maps
```

### Use road-network distances for regional clustering

```bash
python main.py --data data/df_final.csv --output outputs --steps regional --road-network
```

## Outputs

- `outputs/eda/` - exploratory charts and statistical summary tables
- `outputs/national_clusters.png` - national distribution center cluster plot
- `outputs/national_distribution_centers.csv` - national center coordinates
- `outputs/nl_clusters.png` - Nuevo León cluster plot
- `outputs/nl_distribution_centers.csv` - regional center coordinates
- `outputs/baseline_costs_nl.csv` - baseline route cost summary
- `outputs/new_routes_nl.csv` - optimized route assignments and metrics
- `outputs/cluster_map.html` - regional cluster map
- `outputs/comparison_map.html` - baseline vs proposed route map

## Methodology

### Data pipeline

1. Load raw tables for orders, products, clients, order products, and payments
2. Clean and normalize columns, parse dates, and remove invalid rows
3. Merge datasets into a single master table
4. Generate derived features including volumetric weight, customer region, and state coordinates

### Modeling

- **National clustering:** K-Means over state-level coordinates to identify optimal national distribution centers
- **Regional clustering:** K-Medoids for Nuevo León using road-network distances when OSMnx is available

### Route optimization

- Group orders by regional cluster
- Build delivery routes with a heuristic TSP approach
- Calculate fuel consumption, cost, and CO₂ emissions for baseline and proposed routes

## Technologies Used

- Python 3
- Pandas, NumPy
- Scikit-learn, scikit-learn-extra
- Matplotlib, Seaborn
- SciPy
- OSMnx, NetworkX (optional)
- Folium (optional)
- OpenPyXL