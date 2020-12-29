package org.genesys.type;

/**
 * Created by yufeng on 6/4/17.
 * I don't like this one.
 */
public class InitType implements AbstractType {

    public final AbstractType goalType;

    public InitType(AbstractType goal) {
        this.goalType = goal;
    }

    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof InitType)) {
            return false;
        }
        InitType type = (InitType) obj;
        return this.goalType.equals(type.goalType);
    }

    @Override
    public int hashCode() {
        return 5 * this.goalType.hashCode() + 5;
    }
}
