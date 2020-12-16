package org.genesys.models;

/**
 * Created by yufeng on 5/28/17.
 */
public class Quad<T0, T1, T2, T3> {
    public final T0 t0;
    public final T1 t1;
    public final T2 t2;
    public final T3 t3;

    public Quad(T0 t0, T1 t1, T2 t2, T3 t3) {
        this.t0 = t0;
        this.t1 = t1;
        this.t2 = t2;
        this.t3 = t3;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        Quad<?, ?, ?, ?> quad = (Quad<?, ?, ?, ?>) o;

        if (!t0.equals(quad.t0)) return false;
        if (!t1.equals(quad.t1)) return false;
        if (!t2.equals(quad.t2)) return false;
        return t3.equals(quad.t3);
    }

    @Override
    public int hashCode() {
        int result = t0.hashCode();
        result = 31 * result + t1.hashCode();
        result = 31 * result + t2.hashCode();
        result = 31 * result + t3.hashCode();
        return result;
    }
}