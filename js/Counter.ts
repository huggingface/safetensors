export class Counter<T> extends Map<T, number> {
	incr(k: T, incr: number = 1): void {
		if (!this.has(k)) {
			this.set(k, incr);
		} else {
			this.set(k, this.get(k)! + incr);
		}
	}

	/**
	 * Return elements in decreasing count order.
	 * see Python's most_common(n)
	 */
	sorted(): [T, number][] {
		return Array.from(this).sort((a, b) => b[1] - a[1]);
	}
	/**
	 * Returns a new map, sorted.
	 */
	get sortedMap(): Map<T, number> {
		return new Map(this.sorted());
	}

	/**
	 * Initialize from an array.
	 */
	static from<T>(arr: T[]): Counter<T> {
		const counter = new Counter<T>();
		for (const a of arr) {
			counter.incr(a);
		}
		return counter;
	}

	/**
	 * Incr from an array.
	 */
	incrFrom(arr: T[]): void {
		for (const a of arr) {
			this.incr(a);
		}
	}

	/**
	 * Incr from another instance of same Counter
	 */
	add(c: Counter<T>): void {
		for (const [k, v] of c.entries()) {
			this.incr(k, v);
		}
	}

	/**
	 * Total sum of values.
	 */
	total(): number {
		return Array.from(this.values()).reduce((a, b) => a + b, 0);
	}
}
