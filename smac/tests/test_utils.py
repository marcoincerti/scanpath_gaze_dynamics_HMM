import logging
from smac.utils import read_xls_and_structure
from smac.hmm_learn import estimate_hmms_for_all_subjects

# Configura il logger
logging.basicConfig(level=logging.INFO)

def test_rois_estimation():
    logging.info("Reading data from Excel...")
    subject_data = read_xls_and_structure('tests/demodata.xls')
    
    logging.info(f"Subject Data: {subject_data}")
    
    logging.info("Estimating ROIs for all subjects...")
    subject_rois = estimate_hmms_for_all_subjects(subject_data, max_components=[2,3])
    
    logging.info(f"Estimated ROIs: {subject_rois}")