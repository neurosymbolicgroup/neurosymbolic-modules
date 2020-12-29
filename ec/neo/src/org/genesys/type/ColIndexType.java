package org.genesys.type;


/**
 * Created by yufeng on 9/3/17.
 * ColIndex is also an integer, but with different meaning as normal int value.
 */
public class ColIndexType implements AbstractType {
    @Override
    public boolean equals(Object obj) {
        return obj instanceof ColIndexType;
    }

    @Override
    public int hashCode() {
        return 2;
    }

    @Override
    public String toString() {
        return "ColIndex";
    }
}
