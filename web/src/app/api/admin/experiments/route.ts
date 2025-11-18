import { NextRequest, NextResponse } from 'next/server';
import { createClient } from '@/utils/supabase/server';
import { cookies } from 'next/headers';

export async function GET(req: NextRequest) {
  try {
    const cookieStore = await cookies();
    const supabase = createClient(cookieStore);

    const { searchParams } = new URL(req.url);
    const id = searchParams.get('id');

    if (id) {
      const { data, error } = await supabase
        .from('experiments')
        .select(`
          *,
          task:tasks(*)
        `)
        .eq('id', id)
        .single();

      if (error) throw error;
      return NextResponse.json({ experiment: data });
    }

    const { data, error } = await supabase
      .from('experiments')
      .select(`
        *,
        task:tasks(*)
      `)
      .order('created_at', { ascending: false });

    if (error) throw error;

    return NextResponse.json({ experiments: data || [] });
  } catch (err: any) {
    console.error('experiments list error:', err);
    return NextResponse.json(
      { error: err.message || 'unknown error' },
      { status: 500 }
    );
  }
}

export async function POST(req: NextRequest) {
  try {
    const cookieStore = await cookies();
    const supabase = createClient(cookieStore);
    const body = await req.json();

    const { subject_name, xp_human_only, xp_market_mode, xp_task_id } = body;

    if (!subject_name) {
      return NextResponse.json(
        { error: 'subject_name is required' },
        { status: 400 }
      );
    }

    const { data, error } = await supabase
      .from('experiments')
      .insert([{
        subject_name,
        xp_human_only: xp_human_only ?? false,
        xp_market_mode: xp_market_mode || null,
        xp_task_id: xp_task_id || null,
      }])
      .select(`
        *,
        task:tasks(*)
      `)
      .single();

    if (error) throw error;

    return NextResponse.json({ experiment: data });
  } catch (err: any) {
    console.error('experiment creation error:', err);
    return NextResponse.json(
      { error: err.message || 'unknown error' },
      { status: 500 }
    );
  }
}
