package org.genesys.type;

/**
 * Created by yufeng on 9/15/17.
 */
public class BinopBoolType implements AbstractType {

    @Override
    public boolean equals(Object obj) {
        return obj instanceof BinopBoolType;
    }

    @Override
    public int hashCode() {
        return 47;
    }

    @Override
    public String toString() {
        return "BinopBoolType";
    }
}
