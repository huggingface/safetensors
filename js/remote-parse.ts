#! NODE_OPTIONS=--no-warnings npx tsx remote-parse.ts

import { Counter } from "./Counter";

const SINGLE_FILE = "model.safetensors";
const INDEX_FILE = "model.safetensors.index.json";

type FileName = string;

type TensorName = string;
type Dtype =
	| "F64"
	| "F32"
	| "F16"
	| "I64"
	| "I32"
	| "I16"
	| "I8"
	| "U8"
	| "BOOL";

interface TensorInfo {
	dtype: Dtype;
	shape: number[];
	data_offsets: [number, number];
}

type FileHeader = Record<TensorName, TensorInfo> & {
	__metadata__: Record<string, string>;
};

interface IndexJson {
	dtype?: string;
	/// ^there's sometimes a dtype but it looks inconsistent.
	metadata?: Record<string, string>;
	/// ^ why the inconsistency?
	weight_map: Record<TensorName, FileName>;
}

type ShardedMap = Record<FileName, FileHeader>;

const c = console;

async function parseSingleFile(url: URL): Promise<FileHeader> {
	const bufLengthOfHeaderLE = await (
		await fetch(url, {
			headers: {
				Range: "bytes=0-7",
			},
		})
	).arrayBuffer();
	const lengthOfHeader = new DataView(bufLengthOfHeaderLE).getBigUint64(
		0,
		true
	);
	/// ^little-endian
	const header: FileHeader = await (
		await fetch(url, {
			headers: {
				Range: `bytes=8-${7 + Number(lengthOfHeader)}`,
			},
		})
	).json();
	/// no validation for now, we assume it's a valid FileHeader.
	return header;
}

async function parseIndexFile(url: URL): Promise<ShardedMap> {
	const index: IndexJson = await (await fetch(url)).json();
	/// no validation for now, we assume it's a valid IndexJson.

	const shardedMap: ShardedMap = {};
	const filenames = [...new Set(Object.values(index.weight_map))];
	for (const filename of filenames) {
		const singleUrl = new URL(url.toString().replace(INDEX_FILE, filename));
		shardedMap[filename] = await parseSingleFile(singleUrl);
	}
	return shardedMap;
}

async function doesFileExistOnHub(url: URL): Promise<boolean> {
	const res = await fetch(url, {
		method: "HEAD",
		redirect: "manual",
		/// ^do not follow redirects to save some time
	});
	return res.status >= 200 && res.status < 400;
}

function computeNumOfParams(header: FileHeader): number {
	let n = 0;
	for (const [k, v] of Object.entries(header)) {
		if (k === "__metadata__") {
			continue;
		}
		n += (v as TensorInfo).shape.reduce((a, b) => a * b);
	}
	return n;
}

function computeNumOfParamsSharded(shardedMap: ShardedMap): number {
	let n = 0;
	for (const [k, v] of Object.entries(shardedMap)) {
		n += computeNumOfParams(v);
	}
	return n;
}

function computeNumOfParamsByDtype(header: FileHeader): Counter<Dtype> {
	const n = new Counter<Dtype>();
	for (const [k, v] of Object.entries(header)) {
		if (k === "__metadata__") {
			continue;
		}
		n.incr(
			(v as TensorInfo).dtype,
			(v as TensorInfo).shape.reduce((a, b) => a * b)
		);
	}
	return n;
}

function computeNumOfParamsShardedByDtype(
	shardedMap: ShardedMap
): Counter<Dtype> {
	const n = new Counter<Dtype>();
	for (const [k, v] of Object.entries(shardedMap)) {
		n.add(computeNumOfParamsByDtype(v));
	}
	return n;
}

(async () => {
	const modelIds = (
		await (
			await fetch(`https://huggingface.co/api/models?filter=safetensors`)
		).json()
	).map((m) => m.id);

	for (const id of modelIds) {
		c.debug("===", id);

		const singleUrl = new URL(
			`https://huggingface.co/${id}/resolve/main/${SINGLE_FILE}`
		);
		if (await doesFileExistOnHub(singleUrl)) {
			c.info("single-file", singleUrl.toString());
			c.log(computeNumOfParamsByDtype(await parseSingleFile(singleUrl)));
		}

		const indexUrl = new URL(
			`https://huggingface.co/${id}/resolve/main/${INDEX_FILE}`
		);
		if (await doesFileExistOnHub(indexUrl)) {
			c.info("index-file", indexUrl.toString());
			c.log(computeNumOfParamsShardedByDtype(await parseIndexFile(indexUrl)));
		}
	}
})();
