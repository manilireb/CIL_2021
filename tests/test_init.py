from Co_Clustering.Clustering_Coclustering import Clustering_Coclustering
from KNN_Methods.KNN_Basics import KNN_Basic
from KNN_Methods.KNN_WithMeans import KNN_WithMeans
from KNN_Methods.KNN_WithZScore import KNN_WithZScore
from MF_Methods.MF_NMF import MFNMF
from MF_Methods.MF_SVD import MFSVD
from MF_Methods.MF_SVDpp import MFSVDpp
from SlopeOne.slope_one import Slope_One


def test_coclustering():
    Cocl = Clustering_Coclustering()
    assert Cocl.log_file_name == "CoClustering.json"


def test_MNMF():
    NMFBiased = MFNMF(biased=True)
    NMFUnBiased = MFNMF(biased=False)
    assert NMFBiased.log_file_name == "NMFBiased.json"
    assert NMFUnBiased.log_file_name == "NMFUnBiased.json"


def test_svd():
    SVDBiased = MFSVD(biased=True)
    SVDUnBiased = MFSVD(biased=False)
    assert SVDBiased.log_file_name == "SVDBiased.json"
    assert SVDUnBiased.log_file_name == "SVDUnBiased.json"


def test_svdpp():
    SVDpp = MFSVDpp()
    assert SVDpp.log_file_name == "SVDpp.json"


def test_knnbasic():
    KNNBasic_user_cosine = KNN_Basic(sim_name="cosine", user_based=True)
    KNNBasic_user_msd = KNN_Basic(sim_name="msd", user_based=True)
    KNNBasic_user_pearson = KNN_Basic(sim_name="pearson", user_based=True)
    KNNBasic_user_pearson_baseline = KNN_Basic(sim_name="pearson_baseline", user_based=True)

    KNNBasic_item_cosine = KNN_Basic(sim_name="cosine", user_based=False)
    KNNBasic_item_msd = KNN_Basic(sim_name="msd", user_based=False)
    KNNBasic_item_pearson = KNN_Basic(sim_name="pearson", user_based=False)
    KNNBasic_item_pearson_baseline = KNN_Basic(sim_name="pearson_baseline", user_based=False)

    assert KNNBasic_item_cosine.log_file_name == "KNNBasicItemCosine.json"
    assert KNNBasic_item_msd.log_file_name == "KNNBasicItemMsd.json"
    assert KNNBasic_item_pearson.log_file_name == "KNNBasicItemPearson.json"
    assert KNNBasic_item_pearson_baseline.log_file_name == "KNNBasicItemPearsonBaseline.json"

    assert KNNBasic_user_cosine.log_file_name == "KNNBasicUserCosine.json"
    assert KNNBasic_user_msd.log_file_name == "KNNBasicUserMsd.json"
    assert KNNBasic_user_pearson.log_file_name == "KNNBasicUserPearson.json"
    assert KNNBasic_user_pearson_baseline.log_file_name == "KNNBasicUserPearsonBaseline.json"


def test_knnwithmeans():
    KNNWithMeansUserCosine = KNN_WithMeans(sim_name="cosine", user_based=True)
    KNNWithMeansUserMsd = KNN_WithMeans(sim_name="msd", user_based=True)
    KNNWithMeansUserPearson = KNN_WithMeans(sim_name="pearson", user_based=True)
    KNNWithMeansUserPearsonBaseline = KNN_WithMeans(sim_name="pearson_baseline", user_based=True)

    KNNWithMeansItemCosine = KNN_WithMeans(sim_name="cosine", user_based=False)
    KNNWithMeansItemMsd = KNN_WithMeans(sim_name="msd", user_based=False)
    KNNWithMeansItemPearson = KNN_WithMeans(sim_name="pearson", user_based=False)
    KNNWithMeansItemPearsonBaseline = KNN_WithMeans(sim_name="pearson_baseline", user_based=False)

    assert KNNWithMeansItemCosine.log_file_name == "KNNWithMeansItemCosine.json"
    assert KNNWithMeansItemMsd.log_file_name == "KNNWithMeansItemMsd.json"
    assert KNNWithMeansItemPearson.log_file_name == "KNNWithMeansItemPearson.json"
    assert KNNWithMeansItemPearsonBaseline.log_file_name == "KNNWithMeansItemPearsonBaseline.json"

    assert KNNWithMeansUserCosine.log_file_name == "KNNWithMeansUserCosine.json"
    assert KNNWithMeansUserMsd.log_file_name == "KNNWithMeansUserMsd.json"
    assert KNNWithMeansUserPearson.log_file_name == "KNNWithMeansUserPearson.json"
    assert KNNWithMeansUserPearsonBaseline.log_file_name == "KNNWithMeansUserPearsonBaseline.json"


def test_knnwithzscore():
    KNNWithZScoreUserCosine = KNN_WithZScore(sim_name="cosine", user_based=True)
    KNNWithZScoreUserMsd = KNN_WithZScore(sim_name="msd", user_based=True)
    KNNWithZScoreUserPearson = KNN_WithZScore(sim_name="pearson", user_based=True)
    KNNWithZScoreUserPearsonBaseline = KNN_WithZScore(sim_name="pearson_baseline", user_based=True)

    KNNWithZScoreItemCosine = KNN_WithZScore(sim_name="cosine", user_based=False)
    KNNWithZScoreItemMsd = KNN_WithZScore(sim_name="msd", user_based=False)
    KNNWithZScoreItemPearson = KNN_WithZScore(sim_name="pearson", user_based=False)
    KNNWithZScoreItemPearsonBaseline = KNN_WithZScore(sim_name="pearson_baseline", user_based=False)

    assert KNNWithZScoreItemCosine.log_file_name == "KNNWithZScoreItemCosine.json"
    assert KNNWithZScoreItemMsd.log_file_name == "KNNWithZScoreItemMsd.json"
    assert KNNWithZScoreItemPearson.log_file_name == "KNNWithZScoreItemPearson.json"
    assert KNNWithZScoreItemPearsonBaseline.log_file_name == "KNNWithZScoreItemPearsonBaseline.json"

    assert KNNWithZScoreUserCosine.log_file_name == "KNNWithZScoreUserCosine.json"
    assert KNNWithZScoreUserMsd.log_file_name == "KNNWithZScoreUserMsd.json"
    assert KNNWithZScoreUserPearson.log_file_name == "KNNWithZScoreUserPearson.json"
    assert KNNWithZScoreUserPearsonBaseline.log_file_name == "KNNWithZScoreUserPearsonBaseline.json"


def test_slopeone():
    slopeOne = Slope_One()
    assert slopeOne.log_file_name == "SlopeOne.json"
