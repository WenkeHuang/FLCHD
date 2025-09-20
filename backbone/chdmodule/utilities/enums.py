from enum import StrEnum


class Stage(StrEnum):
    TRAIN = "Train"
    VAL = "Val"
    TEST = "Test"


class Source(StrEnum):
    INTERNAL = "Internal"
    EXTERNAL = "External"


class Hospital(StrEnum):
    BJFC = "BJFC"
    GXFY = "GXFY"
    CSSFY = "CSSFY"
    XYYY = "XYYY"


class Task(StrEnum):
    BINARY = "binary"
    MULTICLASS = "multiclass"
    MULTILABEL = "multilabel"


class CLSSetting(StrEnum):
    BINARY = "binary"
    MC3RE = "mc3re"
    MC4ORE = "mc4ore"
    MC6 = "mc6"


class CPTSetting(StrEnum):
    ALLCPT = "AllCPT"
    NOATRIUMFLOW = "NoAtriumFlow"
    NOCHAMBERNUM = "NoChamberNum"
