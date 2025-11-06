import { z } from 'zod';

type Env = z.infer<typeof envSchema>;
const envSchema = z.object({
    STORE_MODE: z.enum(['hotel', 'airline'], {
        message: 'STORE_MODE must be either "hotel" or "airline"'
    }),
    NEXT_PUBLIC_API_BASE: z.string().url({
        message: 'NEXT_PUBLIC_API_BASE must be a valid URL (e.g., http://localhost:3000)'
    }),
    NEXT_PUBLIC_APP_ENV: z.enum(['dev', 'prod'], {
        message: 'NEXT_PUBLIC_APP_ENV must be either "dev" or "prod"'
    }),
});

// parse and validate env at module load, fail fast with descriptive errors
const parseEnv = (): Env => {
    const result = envSchema.safeParse({
        STORE_MODE: process.env.STORE_MODE,
        NEXT_PUBLIC_API_BASE: process.env.NEXT_PUBLIC_API_BASE,
        NEXT_PUBLIC_APP_ENV: process.env.NEXT_PUBLIC_APP_ENV,
    });
    if (!result.success) {
        const errors = result.error.issues.map((err) => `${err.path.join('.')}: ${err.message}`).join('\n');
        throw new Error(`Environment validation failed:\n${errors}`);
    }
    return result.data;
};

export const config: Env = parseEnv();
