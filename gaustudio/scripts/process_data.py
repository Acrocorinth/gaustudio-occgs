from typing import Optional

import click


@click.command()
@click.option("--dataset", "-d", type=str, default="waymo", help="Dataset name (polycam, mvsnet, nerf, scannet, waymo)")
@click.option("--source_path", "-s", required=True, help="Path to the dataset")
@click.option("--output_dir", "-o", required=True, help="Path to the output directory")
@click.option("--init", default="loftr", type=str, help="Initializer name (colmap, loftr, dust3r, mvsplat, midas)")
@click.option("--overwrite", help="Overwrite existing files", is_flag=True)
@click.option("--w_mask", "-w", is_flag=True, help="Use mask")
@click.option("--resolution", "-r", default=1, type=int, help="Resolution")
def main(
    dataset: str,
    source_path: Optional[str],
    output_dir: Optional[str],
    init: str,
    overwrite: bool,
    w_mask: bool,
    resolution: int,
) -> None:
    """
    Main function to run the pipeline.

    Args:
        dataset (str): Name of the dataset.
        source_path (Optional[str]): Path to the dataset.
        output_dir (Optional[str]): Path to the output directory.
        init (str): Name of the initializer.
        overwrite (bool): Whether to overwrite existing files.
        with_mask (bool): Whether to use mask.
    """
    from gaustudio import datasets, models
    from gaustudio.pipelines import initializers

    dataset_config = {
        "name": dataset,
        "source_path": source_path,
        "w_mask": w_mask,
        "camera_number": 1,
    }

    dataset_instance = datasets.make(dataset_config)
    dataset_instance.all_cameras = [_camera.downsample_scale(resolution) for _camera in dataset_instance.all_cameras]
    pcd = models.make("general_pcd")
    initializer_config = {"name": init, "workspace_dir": output_dir}
    initializer_instance = initializers.make(initializer_config)

    initializer_instance(pcd, dataset_instance, overwrite=overwrite)


if __name__ == "__main__":
    main()
