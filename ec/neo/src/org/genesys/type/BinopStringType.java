package org.genesys.type;

/**
 * Created by yufeng on 9/15/17.
 */
public class BinopStringType implements AbstractType {

    @Override
    public boolean equals(Object obj) {
        return obj instanceof BinopStringType;
    }

    @Override
    public int hashCode() {
        return 611;
    }

    @Override
    public String toString() {
        return "BinopStringType";
    }
}
