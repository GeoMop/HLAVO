from hlavo.ingress.moist_profile.extract.main import main


if __name__ == '__main__':
    """
    Tests
    - reading downloaded CSV files from xpert.nz
    - creating dataframe according to zarr_fuse schema
    - creating zarr_fuse local storage
    """
    main(source_dir="20260201T205548_dataflow_grab")