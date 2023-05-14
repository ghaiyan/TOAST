import numpy as np
import argparse

def get_res_string(res):
	"""Converts resolution in bp to string (e.g. 10kb)"""
	res_kb = int(res/1000)
	if res_kb < 1000:
		return str(res_kb) + "kb"
	else:
		return str(res_kb/1000) + "mb"	

def normalize(chrom1, chrom2, rawpath, krpath1, krpath2, res, outpath):
	kr1 = np.loadtxt(krpath1)
	if krpath2 is None:
		kr2 = kr1
	else:
		kr2 = np.loadtxt(krpath2)
	with open(rawpath) as raw:
		with open(outpath, "w") as out:
			for line in raw:
				line = line.split()
				loc1 = line[0]
				loc2 = line[1]
				norm1 = kr1[int(int(loc1)/res)]
				norm2 = kr2[int(int(loc2)/res)]
				if not np.isnan(norm1) and not np.isnan(norm2):
					out.write("\t".join((chrom1, loc1, str(int(loc1) + res), chrom2, loc2, str(int(loc2) + res), str(float(line[2])/(norm1 * norm2)))) + "\n")
		out.close()
	raw.close()

def normalize_intra(hic_id, res, chrom):
	res_string = get_res_string(res)
	rawpath = "chr{}_{}.RAWobserved".format(chrom, res_string)
	krpath = "chr{}_{}.KRnorm".format(chrom, res_string)
	outpath = "{}_{}_{}.bed".format(hic_id, chrom, res_string)
	chromstring = "chr" + chrom
	normalize(chromstring, chromstring, rawpath, krpath, None, res, outpath)


def normalize_inter(hic_id, res, chrom1, chrom2):
	res_string = get_res_string(res)

	rawpath = "chr{}_{}_{}.RAWobserved".format(chrom1, chrom2, res_string)
	krpath1 = "chr{}_{}.KRnorm".format(chrom1, res_string)
	krpath2 = "chr{}_{}.KRnorm".format(chrom2, res_string)
	outpath = "{}_{}_{}_{}.bed".format(hic_id, chrom1, chrom2, res_string)
	chromstring1 = "chr" + chrom1
	chromstring2 = "chr" + chrom2
	normalize(chromstring1, chromstring2, rawpath, krpath1, krpath2, res, outpath)

def main():
	parser = argparse.ArgumentParser(description="Normalize Hi-C files using Knight-Ruiz method.")
	parser.add_argument("hic_id", help="e.g. GM12878")
	parser.add_argument("res", type=int, help="resolution (bp)")
	parser.add_argument("chrom1", help="first chromosome (e.g. 1)")
	parser.add_argument("--chrom2", help="second chromosome (e.g. 2)")
	args = parser.parse_args()

	if args.chrom2 is None:
		normalize_intra(args.hic_id, args.res, args.chrom1)
	else:
		normalize_inter(args.hic_id, args.res, args.chrom1, args.chrom2)

if __name__ == "__main__":
	main()
