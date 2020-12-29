package org.genesys.type;

/**
 * Created by yufeng on 5/31/17.
 */
public class BoolType implements AbstractType {
    @Override
    public boolean equals(Object obj) {
        return obj instanceof BoolType;
    }

    @Override
    public int hashCode() {
        return 0;
    }

    @Override
    public String toString() {
        return "Boolean";
    }
}
