import tools.axis_chipcheck as cc

VAPIX_A8 = (
    "root.Properties.System.Architecture=aarch64\n"
    "root.Properties.System.Soc=Axis Artpec-8\n"
    "root.Properties.System.SerialNumber=ABISCRUBBED\n"
)
VAPIX_A7 = (
    "root.Properties.System.Architecture=armv7hf\n"
    "root.Properties.System.Soc=Axis Artpec-7\n"
)
VAPIX_NONE = "root.Properties.System.Architecture=armv7hf\n"


def test_detect_artpec_8():
    assert cc.detect_artpec(VAPIX_A8) == 8


def test_detect_artpec_7():
    assert cc.detect_artpec(VAPIX_A7) == 7


def test_detect_artpec_absent():
    assert cc.detect_artpec(VAPIX_NONE) is None


def test_classify_8_is_on_camera():
    r = cc.classify(8)
    assert r["on_camera_viable"] is True
    assert r["recommendation"] == "on-camera"


def test_classify_9_is_on_camera():
    assert cc.classify(9)["on_camera_viable"] is True


def test_classify_7_is_fallback():
    r = cc.classify(7)
    assert r["on_camera_viable"] is False
    assert r["recommendation"] == "jetson-fallback"


def test_classify_unknown_is_fallback():
    assert cc.classify(None)["recommendation"] == "jetson-fallback"
