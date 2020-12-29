package org.genesys.type;


/**
 * Created by yufeng on 5/31/17.
 */
public class AggrType implements AbstractType {
    @Override
    public boolean equals(Object obj) {
        return obj instanceof AggrType;
    }

    @Override
    public int hashCode() {
        return 1;
    }

    @Override
    public String toString() {
        return "AggrType";
    }
}
