#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import safetensors.numpy

import safetensors


__author__ = "KOLANICH"
__copyright__ = "Public domain"
__license__ = "Unlicense"

__reuse__ = """SPDX-FileCopyrightText: Uncopyrightable
SPDX-License-Identifier: Unlicense
"""

thisDir = Path(".").absolute()


ambigiousTypes = set("LPQlpq")  # these letters result into the same arrays


def makeTypeCodes():
	typeCodes = "".join(
		sorted(set("".join(np.typecodes[el] for el in ("Complex", "AllInteger", "AllFloat"))))
	)
	typeCodes = "".join(set(typeCodes) - ambigiousTypes)
	return typeCodes


def getPanicExceptionType():
	try:
		safetensors.safetensors_rust.serialize(
			{"tensor_name": {"dtype": "test", "shape": [0, 0], "data": b""}}, None
		)
	except BaseException as ex:  # pylint:disable=broad-except
		return ex.__class__
	return None


PanicException = getPanicExceptionType()


def makeNumpyDict(typeCodes) -> dict:
	npDict = {}
	for t in typeCodes:
		a = np.array(range(-3, 3), dtype="<" + t)
		try:
			safetensors.numpy.save({"a": a})  # testing if the type safetensors-serializeable
		except PanicException:  # pylint:disable=broad-except
			print(t, a.dtype)
		else:
			npDict[t] = a
	return npDict


EXT = ".safetensors"


def dumpFile(name: str, data: bytes):
	fn = name + EXT
	(thisDir / fn).write_bytes(data)
	(thisDir / (fn + ".license")).write_text(__reuse__)


def main():
	typeCodes = makeTypeCodes()
	npDict = makeNumpyDict(typeCodes)
	metaDict = {"test": "a"}
	dumpFile("overall", safetensors.numpy.save(npDict, metaDict))

	for k, v in npDict.items():
		dumpFile(k, safetensors.numpy.save({k: v}, metaDict))


if __name__ == "__main__":
	main()
