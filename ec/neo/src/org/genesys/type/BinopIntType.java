package org.genesys.type;

/**
 * Created by yufeng on 9/15/17.
 */
public class BinopIntType implements AbstractType {

    @Override
    public boolean equals(Object obj) {
        return obj instanceof BinopIntType;
    }

    @Override
    public int hashCode() {
        return 61;
    }

    @Override
    public String toString() {
        return "BinopIntType";
    }
}
