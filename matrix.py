import sys
import numpy as np
import argparse
from decimal import *


class ChromParameters(object):
    """Basic information on chromosome, inferred from input file"""

    def __init__(self, minPos, maxPos, res, name):
        self.minPos = minPos  # minimum genomic coordinate
        self.maxPos = maxPos  # maximum genomic coordinate
        self.res = res  # resolution (bp)
        self.name = name  # e.g. "chr22"

    def getLength(self):
        """Number of possible loci"""
        return int((self.maxPos - self.minPos)/self.res) + 1

    def getAbsoluteIndex(self, genCoord):
        """Converts genomic coordinate into absolute index. Absolute indexing includes empty (zero) points."""
        if genCoord < self.minPos or genCoord > self.maxPos + self.res:
            return None
        else:
            return int((genCoord - self.minPos)/self.res)

    def getGenCoord(self, abs_index):
        """Converts absolute index into genomic coordinate"""
        return self.minPos + self.res * abs_index

    def reduceRes(self, resRatio):
        """Creates low-res version of this chromosome"""
        lowRes = self.res * resRatio
        # approximate at low resolution
        lowMinPos = (self.minPos/lowRes)*lowRes
        lowMaxPos = (self.maxPos/lowRes)*lowRes
        return ChromParameters(lowMinPos, lowMaxPos, lowRes, self.name)


def chromFromBed(path, minPos=None, maxPos=None):
    """Initialize ChromParams from intrachromosomal file in BED format"""
    overall_minPos = sys.float_info.max
    overall_maxPos = 0
    print("Scanning intrachromosomal{}".format(path))
    with open(path) as infile:
        for i, line in enumerate(infile):
            line = line.strip().split()
            if minPos is None or maxPos is None:
                pos1 = int(line[1])
                pos2 = int(line[4])
                if minPos is None:
                    curr_minPos = min((pos1, pos2))
                    if curr_minPos < overall_minPos:
                        overall_minPos = curr_minPos
                if maxPos is None:
                    curr_maxPos = max((pos1, pos2))
                    if curr_maxPos > overall_maxPos:
                        overall_maxPos = curr_maxPos
            if i == 0:
                name = line[0]
                res = (int(line[2]) - pos1)
        infile.close()
    minPos = int(np.floor(float(overall_minPos)/res)) * res  # round
    maxPos = int(np.ceil(float(overall_maxPos)/res)) * res
    return ChromParameters(minPos, maxPos, res, name)


def matrixFromBed(path):
    """Contact Matrix from intrachromosomal BED file."""
    chrom = chromFromBed(path)
    start = chrom.minPos
    end = chrom.maxPos
    length = chrom.getLength()
    matrix = np.zeros((length, length))

    # add loci
    with open(path) as listFile:
        for line in listFile:
            line = line.strip().split()
            pos1 = int(line[1])
            pos2 = int(line[4])
            if pos1 >= start and pos1 <= end and pos2 >= start and pos2 <= end:
                abs_index1 = chrom.getAbsoluteIndex(pos1)
                abs_index2 = chrom.getAbsoluteIndex(pos2)
                matrix[abs_index1][abs_index2] = Decimal(line[6])
                matrix[abs_index2][abs_index1] = Decimal(line[6])
        listFile.close()
    np.savetxt("{}_{}.hic".format(chrom.name, chrom.res), matrix)


def interChromFromBed(path):
    """Initialize ChromParams from interchromosomal file in BED format"""
    chrom1_maxPos = 0
    chrom2_maxPos = 0
    print("Scanning interchromosomal{}".format(path))
    with open(path) as infile:
        for i, line in enumerate(infile):
            line = line.strip().split()
            chrom1pos = int(line[1])
            chrom2pos = int(line[4])
            if chrom1pos > chrom1_maxPos:
                chrom1_maxPos = chrom1pos
            if chrom2pos > chrom2_maxPos:
                chrom2_maxPos = chrom2pos
            if i == 0:
                name1 = line[0]
                name2 = line[3]
                res = (int(line[2]) - int(line[1]))
        infile.close()
    return chrom1_maxPos, chrom2_maxPos, res, name1, name2


def interMatrixFromBed(path):
    chrom1_maxPos, chrom2_maxPos, res, name1, name2 = interChromFromBed(path)
    length1 = int(chrom1_maxPos / res) + 1
    length2 = int(chrom2_maxPos / res) + 1
    matrix = np.zeros((length1, length2))
    with open(path) as listFile:
        for line in listFile:
            line = line.strip().split()
            pos1 = int(line[1])
            pos2 = int(line[4])
            abs_index1 = int(pos1 / res)
            abs_index2 = int(pos2 / res)
            matrix[abs_index1][abs_index2] = Decimal(line[6])
        listFile.close()
    np.savetxt("{}_{}_{}.hic".format(name1, name2, res), matrix)


def distinguish(path):
    if len(path.split('_')) == 3:
        print("intrachromosomal")
        matrixFromBed(path)
    elif len(path.split('_')) == 4:
        print("interchromosomal")
        interMatrixFromBed(path)
    else:
        print("error")


def main():
    parser = argparse.ArgumentParser(description="hypergraph demo")
    parser.add_argument("path", help="path to intrachromosomal Hi-C BED file")
    args = parser.parse_args()

    distinguish(args.path)


if __name__ == "__main__":
    main()
