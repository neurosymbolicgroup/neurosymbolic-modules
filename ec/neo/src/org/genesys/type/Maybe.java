package org.genesys.type;

/**
 * Created by yufeng on 5/31/17.
 */
public class Maybe<T> {
    private final T t;

    public Maybe(T t) {
        this.t = t;
    }

    public Maybe() {
        this.t = null;
    }

    public T get() {
        if (!this.has()) {
            throw new RuntimeException("Invalid access!");
        }
        return this.t;
    }

    public boolean has() {
        return this.t != null;
    }
}