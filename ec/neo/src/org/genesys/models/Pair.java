package org.genesys.models;

/**
 * Created by yufeng on 5/26/17.
 */
public class Pair<T0, T1> {
    public final T0 t0;
    public final T1 t1;

    public Pair(T0 t0, T1 t1) {
        this.t0 = t0;
        this.t1 = t1;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        Pair<?, ?> pair = (Pair<?, ?>) o;

        if (!t0.equals(pair.t0)) return false;
        return t1.equals(pair.t1);
    }

    @Override
    public int hashCode() {
        int result = t0.hashCode();
        result = 31 * result + t1.hashCode();
        return result;
    }

    @Override
    public String toString() {
        return "Pair{" +
                "t0=" + t0 +
                ", t1=" + t1 +
                '}';
    }
}
