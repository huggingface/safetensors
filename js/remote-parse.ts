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
	| "BF16"
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

type ShardedHeaders = Record<FileName, FileHeader>;

type ParseFromRepo =
	| {
			sharded: false;
			header: FileHeader;
	  }
	| {
			sharded: true;
			index: IndexJson;
			headers: ShardedHeaders;
	  };

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

async function parseIndexFile(
	url: URL
): Promise<{ index: IndexJson; headers: ShardedHeaders }> {
	const index: IndexJson = await (await fetch(url)).json();
	/// no validation for now, we assume it's a valid IndexJson.

	const shardedMap: ShardedHeaders = {};
	const filenames = [...new Set(Object.values(index.weight_map))];
	await Promise.all(
		filenames.map(async (filename) => {
			const singleUrl = new URL(url.toString().replace(INDEX_FILE, filename));
			shardedMap[filename] = await parseSingleFile(singleUrl);
		})
	);
	return { index, headers: shardedMap };
}

async function doesFileExistOnHub(url: URL): Promise<boolean> {
	const res = await fetch(url, {
		method: "HEAD",
		redirect: "manual",
		/// ^do not follow redirects to save some time
	});
	return res.status >= 200 && res.status < 400;
}

async function parseFromModelRepo(id: string): Promise<ParseFromRepo> {
	const singleUrl = new URL(
		`https://huggingface.co/${id}/resolve/main/${SINGLE_FILE}`
	);
	const indexUrl = new URL(
		`https://huggingface.co/${id}/resolve/main/${INDEX_FILE}`
	);
	if (await doesFileExistOnHub(singleUrl)) {
		return {
			sharded: false,
			header: await parseSingleFile(singleUrl),
		};
	} else if (await doesFileExistOnHub(indexUrl)) {
		return {
			sharded: true,
			...(await parseIndexFile(indexUrl)),
		};
	} else {
		throw new Error("model id does not contain safetensors weights");
	}
}

function computeNumOfParams(header: FileHeader): number {
	let n = 0;
	for (const [k, v] of Object.entries(header)) {
		if (k === "__metadata__") {
			continue;
		}
		if ((v as TensorInfo).shape.length === 0) {
			continue;
		}
		n += (v as TensorInfo).shape.reduce((a, b) => a * b);
	}
	return n;
}

function computeNumOfParamsSharded(shardedMap: ShardedHeaders): number {
	let n = 0;
	for (const [k, v] of Object.entries(shardedMap)) {
		n += computeNumOfParams(v);
	}
	return n;
}

function computeNumOfParamsByDtypeSingleFile(
	header: FileHeader
): Counter<Dtype> {
	const n = new Counter<Dtype>();
	for (const [k, v] of Object.entries(header)) {
		if (k === "__metadata__") {
			continue;
		}
		if ((v as TensorInfo).shape.length === 0) {
			continue;
		}
		n.incr(
			(v as TensorInfo).dtype,
			(v as TensorInfo).shape.reduce((a, b) => a * b)
		);
	}
	return n;
}

function computeNumOfParamsByDtypeSharded(
	shardedMap: ShardedHeaders
): Counter<Dtype> {
	const n = new Counter<Dtype>();
	for (const [k, v] of Object.entries(shardedMap)) {
		n.add(computeNumOfParamsByDtypeSingleFile(v));
	}
	return n;
}

function computeNumOfParamsByDtype(parse: ParseFromRepo): Counter<Dtype> {
	if (parse.sharded) {
		return computeNumOfParamsByDtypeSharded(parse.headers);
	} else {
		return computeNumOfParamsByDtypeSingleFile(parse.header);
	}
}

function formatCounter<T>(counter: Counter<T>): string {
	const inner = [...counter.entries()]
		.map(([k, v]) => `'${k}' => ${v}`)
		.join(", ");
	return `{ ${inner} }`;
}

(async () => {
	const modelIds = (
		await (
			await fetch(
				`https://huggingface.co/api/models?filter=safetensors&sort=downloads&direction=-1&limit=100`
			)
		).json()
	).map((m) => m.id);

	c.debug("model | safetensors | params");
	c.debug("--- | --- | ---");
	for (const id of modelIds) {
		try {
			const p = await parseFromModelRepo(id);
			c.debug(
				[
					id,
					p.sharded ? "index-file" : "single-file",
					formatCounter(computeNumOfParamsByDtype(p)),
				].join(" | ")
			);
		} catch (err) {
			c.debug([id, "error", err.message].join(" | "));
		}
	}
	process.exit();
})();
