package org.genesys.ml;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Utils {
	public static <T> String toString(List<T> ts) {
		StringBuilder sb = new StringBuilder();
		sb.append("[");
		for(T t : ts) {
			sb.append(t).append(", ");
		}
		if(ts.size() > 0) {
			sb.delete(sb.length()-2, sb.length());
		}
		sb.append("]");
		return sb.toString();
	}
	public static class Counter<T> {
		private final Map<T,Integer> counts = new HashMap<T,Integer>();
		public void increment(T t) {
			if(!this.counts.containsKey(t)) {
				this.counts.put(t, 1);
			} else {
				this.counts.put(t, this.counts.get(t)+1);
			}
		}
		public int getCount(T t) {
			return this.counts.getOrDefault(t, 0);
		}
	}
}
