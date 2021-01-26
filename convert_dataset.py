from os.path import join

from dataset_utils.NoduleDatasetPaths import NoduleDatasetPaths
from dataset_utils.raw.NoduleRawConverter import NoduleRawConverter
from dataset_utils.raw.NoduleRawGenerator import NoduleRawGenerator

gen_name = 'test'

paths = NoduleDatasetPaths()
dataset_out_path = join(paths.generated_root, gen_name)


raw_generator = NoduleRawGenerator(paths.raw_images, paths.raw_annotations)
converter = NoduleRawConverter(raw_generator, dataset_out_path)

converter.convert_and_save_cropped(
    crops_per_region=3,
    bg_crops=7,
    crop_min_size=150,
    random_seed=24,
    annotate_images=False
)