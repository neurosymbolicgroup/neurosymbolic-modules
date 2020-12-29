package org.genesys.type;

/**
 * Created by yufeng on 6/4/17.
 */
public class PairType implements AbstractType {
    public final AbstractType firstType;
    public final AbstractType secondType;

    public PairType(AbstractType firstType, AbstractType secondType) {
        this.firstType = firstType;
        this.secondType = secondType;
    }

    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof PairType)) {
            return false;
        }
        PairType type = (PairType) obj;
        return this.firstType.equals(type.firstType) && this.secondType.equals(type.secondType);
    }

    @Override
    public int hashCode() {
        return 5 * (5 * this.firstType.hashCode() + this.secondType.hashCode()) + 2;
    }

    @Override
    public String toString() {
        return "(" + this.firstType.toString() + ", " + this.secondType.toString() + ")";
    }
}