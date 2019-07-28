from miur_daad_dataset_pipeline import build, build_test

def test_build():
    build_test("test_dataset")
    build("test_dataset")
    build("test_dataset")