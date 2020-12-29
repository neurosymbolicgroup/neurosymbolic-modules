package org.genesys.type;

/**
 * Created by yufeng on 5/31/17.
 */
public class ListType implements AbstractType {
    public final AbstractType type;

    public ListType(AbstractType type) {
        this.type = type;
    }

    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof ListType)) {
            return false;
        }
        return this.type.equals(((ListType) obj).type);
    }

    @Override
    public int hashCode() {
        return 5 * this.type.hashCode() + 1;
    }

    @Override
    public String toString() {
        return "List<" + this.type.toString() + ">";
    }
}
