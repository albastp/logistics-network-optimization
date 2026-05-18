import argparse
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Logistics Optimization Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--data", type=str, default="data/df_final.csv",
        help="Path to the processed dataset CSV.",
    )
    parser.add_argument(
        "--raw-data-dir", type=str, default=None,
        help="Directory with raw Excel files. If provided, rebuilds the dataset from scratch.",
    )
    parser.add_argument(
        "--output", type=str, default="outputs",
        help="Directory for all output files.",
    )
    parser.add_argument(
        "--steps", nargs="+",
        choices=["eda", "national", "regional", "routing", "maps", "all"],
        default=["all"],
        help=(
            "Pipeline steps to run:\n"
            "  eda         - Exploratory data analysis (charts, ANOVA, chi-square)\n"
            "  national    - National K-Means clustering (7 centers)\n"
            "  regional    - Nuevo León regional clustering (5 centers)\n"
            "  routing     - Route optimization + cost comparison\n"
            "  maps        - Interactive Folium HTML maps\n"
            "  all         - Run every step"
        ),
    )
    parser.add_argument(
        "--road-network", action="store_true",
        help="Use OSMnx road-network distances for NL clustering (requires osmnx + scikit-learn-extra).",
    )
    parser.add_argument(
        "--graphml", type=str, default=None,
        help="Path to precomputed NL road graph (.graphml). Skips downloading if provided.",
    )
    parser.add_argument(
        "--dist-matrix", type=str, default=None,
        help="Path to precomputed NL distance matrix (.npy). Skips rebuilding if provided.",
    )
    parser.add_argument(
        "--no-ortools", action="store_true",
        help="Use nearest-neighbor heuristic instead of OR-Tools TSP (faster, lower quality).",
    )
    parser.add_argument(
        "--osm-routing", action="store_true",
        help="Use OSMnx road distances for route cost computation (requires osmnx).",
    )
    parser.add_argument(
        "--n-national", type=int, default=7,
        help="Number of national distribution centers (default: 7).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    steps = set(args.steps)
    run_all = "all" in steps

    os.makedirs(args.output, exist_ok=True)

    if args.raw_data_dir:
        from src.data_processing.build_dataset import build_master_dataset
        print(f"Building master dataset from {args.raw_data_dir}...")
        df = build_master_dataset(args.raw_data_dir, output_path=args.data)
    else:
        from src.data_processing.build_dataset import load_processed
        print(f"Loading processed dataset from {args.data}...")
        if not os.path.exists(args.data):
            print(f"ERROR: {args.data} not found. Use --raw-data-dir to build it first.")
            sys.exit(1)
        df = load_processed(args.data)

    print(f"Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns\n")

    df_nl_clustered = None

    if run_all or "eda" in steps:
        from src.eda.statistical_analysis import run_full_eda
        print("=" * 50)
        print("[1/5] Exploratory Data Analysis")
        print("=" * 50)
        run_full_eda(df, output_dir=f"{args.output}/eda")

    if run_all or "national" in steps:
        from src.modeling.kmeans_national import run_national_clustering
        print("\n" + "=" * 50)
        print("[2/5] National K-Means Clustering")
        print("=" * 50)
        _, centers_national, _ = run_national_clustering(
            df, output_dir=args.output, n_clusters=args.n_national
        )
        print(f"National centers: {len(centers_national)}")

    if run_all or "regional" in steps:
        from src.modeling.kmedoids_nl import run_nl_clustering
        print("\n" + "=" * 50)
        print("[3/5] Nuevo León Regional Clustering")
        print("=" * 50)
        df_nl_clustered, centers_nl, _ = run_nl_clustering(
            df,
            output_dir=args.output,
            use_road_network=args.road_network,
            graphml_path=args.graphml,
        )
        print(f"NL centers: {len(centers_nl)}")

    if run_all or "routing" in steps:
        from src.routing.route_optimizer import run_routing_pipeline
        print("\n" + "=" * 50)
        print("[4/5] Route Optimization")
        print("=" * 50)
        if df_nl_clustered is None:
            from src.modeling.kmedoids_nl import run_nl_clustering
            df_nl_clustered, centers_nl, _ = run_nl_clustering(
                df, output_dir=args.output,
                use_road_network=args.road_network,
                graphml_path=args.graphml,
            )
        comparison = run_routing_pipeline(
            df_nl_clustered,
            output_dir=args.output,
            use_osm=args.osm_routing,
            use_ortools=not args.no_ortools,
            graphml_path=args.graphml,
            generate_maps=False,
        )
        savings = comparison["baseline"]["gasto_gasolina_total"] - comparison["proposed"]["gasto_gasolina_total"]
        co2_saved = comparison["baseline"]["co2_total_kg"] - comparison["proposed"]["co2_total_kg"]
        print(f"\n  Fuel savings: ${savings:,.2f}")
        print(f"  CO₂ reduction: {co2_saved:,.1f} kg")

    if run_all or "maps" in steps:
        import pandas as pd
        from src.visualization.maps import generate_all_maps
        print("\n" + "=" * 50)
        print("[5/5] Generating Interactive Maps")
        print("=" * 50)
        clustered_path = f"{args.output}/nl_orders_clustered.csv"
        centers_path = f"{args.output}/nl_distribution_centers.csv"
        baseline_path = f"{args.output}/baseline_costs_nl.csv"
        proposed_path = f"{args.output}/new_routes_nl.csv"
        missing = [p for p in [clustered_path, centers_path, baseline_path, proposed_path]
                   if not os.path.exists(p)]
        if missing:
            print(f"  Skipping maps — missing files: {missing}")
            print("  Run with --steps regional routing first.")
        else:
            generate_all_maps(
                df_clustered=pd.read_csv(clustered_path),
                centers=pd.read_csv(centers_path),
                df_baseline=pd.read_csv(baseline_path),
                df_proposed=pd.read_csv(proposed_path),
                output_dir=args.output,
            )

    print(f"\nAll outputs saved to: {args.output}/")


if __name__ == "__main__":
    main()
