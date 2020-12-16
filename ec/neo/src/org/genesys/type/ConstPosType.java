package org.genesys.type;


/**
 * Created by yufeng on 5/31/17.
 */
public class ConstPosType implements AbstractType {
    @Override
    public boolean equals(Object obj) {
        return obj instanceof ConstPosType;
    }

    @Override
    public int hashCode() {
        return 111 * this.hashCode() + 11;
    }

    @Override
    public String toString() {
        return "Constant Pos";
    }
}