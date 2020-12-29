package org.genesys.models;

/**
 * Created by yufeng on 5/28/17.
 */
public class Trio<T0, T1, T2> {
    public final T0 t0;
    public final T1 t1;
    public final T2 t2;

    public Trio(T0 t0, T1 t1, T2 t2) {
        this.t0 = t0;
        this.t1 = t1;
        this.t2 = t2;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        Trio<?, ?, ?> trio = (Trio<?, ?, ?>) o;

        if (!t0.equals(trio.t0)) return false;
        if (!t1.equals(trio.t1)) return false;
        return t2.equals(trio.t2);
    }

    @Override
    public int hashCode() {
        int result = t0.hashCode();
        result = 31 * result + t1.hashCode();
        result = 31 * result + t2.hashCode();
        return result;
    }
}