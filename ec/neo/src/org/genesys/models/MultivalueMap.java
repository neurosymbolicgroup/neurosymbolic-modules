package org.genesys.models;

import java.util.*;

/**
 * Created by yufeng on 5/26/17.
 */
public class MultivalueMap<K, V> {
    private final Map<K, List<V>> map = new HashMap<K, List<V>>();

    public void add(K k, V v) {
        if (!this.map.containsKey(k)) {
            this.map.put(k, new ArrayList<V>());
        }
        this.map.get(k).add(v);
    }

    public List<V> get(K k) {
        return this.map.containsKey(k) ? this.map.get(k) : new ArrayList<V>();
    }

    public Set<K> keySet() {
        return this.map.keySet();
    }

    public int size() {
        return this.map.size();
    }

    public boolean containsKey(K k) {
        return this.map.containsKey(k);
    }
}
