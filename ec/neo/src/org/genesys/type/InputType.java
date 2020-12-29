package org.genesys.type;

/**
 * Created by yufeng on 6/4/17.
 */
public class InputType implements AbstractType {

    private final int index;

    private AbstractType type;

    public InputType(int idx, AbstractType t) {
        this.index = idx;
        type = t;
    }

    public int getIndex() {
        return index;
    }

    public AbstractType getType() {
        return type;
    }

    @Override
    public String toString() {
        return "InputType{" +
                "index=" + index +
                '}';
    }
}
