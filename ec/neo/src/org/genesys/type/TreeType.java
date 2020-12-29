package org.genesys.type;

/**
 * Created by yufeng on 5/31/17.
 */
public class TreeType implements AbstractType {
    public final AbstractType type;

    public TreeType(AbstractType type) {
        this.type = type;
    }

    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof TreeType)) {
            return false;
        }
        return this.type.equals(((TreeType) obj).type);
    }

    @Override
    public int hashCode() {
        return 7 * this.type.hashCode() + 1;
    }

    @Override
    public String toString() {
        return "Tree{" + this.type.toString() + "}";
    }
}
